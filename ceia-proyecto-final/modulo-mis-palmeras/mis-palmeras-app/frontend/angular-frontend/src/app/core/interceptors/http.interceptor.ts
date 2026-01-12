import { HttpInterceptorFn } from "@angular/common/http";
import { environment } from "../../../environments/environment";

export const apiInterceptor: HttpInterceptorFn = (req, next) => {
  if (req.url.startsWith(environment.keycloak.authority)) {
    return next(req); // Bypass the interceptor for requests to the auth server
  }
  const apiReq = req.clone({ url: `${environment.backend.apiBaseUrl}${req.url}` });
  return next(apiReq);
};
