import { HttpInterceptorFn } from "@angular/common/http";

export const apiInterceptor: HttpInterceptorFn = (req, next) => {
  if (req.url.startsWith('http://localhost:7000')) {
    return next(req); // Bypass the interceptor for requests to the auth server
  }
  const apiReq = req.clone({ url: `http://localhost:8000/apiv1${req.url}` });
  return next(apiReq);
};
