import { inject } from "@angular/core";
import { HttpInterceptorFn } from "@angular/common/http";
import { OidcSecurityService } from "angular-auth-oidc-client";
import { catchError, firstValueFrom, switchMap } from "rxjs";

export const tokenInterceptor: HttpInterceptorFn = (req, next) => {
  const oidcSecurityService = inject(OidcSecurityService);

  if (req.url.startsWith('http://localhost:7000')) {
    return next(req); // Bypass the interceptor for requests to the auth server
  }

  console.log('Token interceptor triggered for request:', req);

  return oidcSecurityService.getAccessToken().pipe(
    switchMap((token) => {
      const request = req.clone({
        setHeaders: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      });
      console.log('Request with token:', request);

      return next(request);
    }),
    catchError((err) => {
      console.error('Error fetching token:', err);
      return next(req);
    })
  );
};
