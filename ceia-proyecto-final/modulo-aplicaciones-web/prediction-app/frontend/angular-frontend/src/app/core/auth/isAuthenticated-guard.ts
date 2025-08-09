import { inject, Signal } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { AuthenticatedResult, OidcSecurityService } from 'angular-auth-oidc-client';

export const isAuthenticatedGuard: CanActivateFn = (route, state) => {
  const oidcSecurityService = inject(OidcSecurityService);
  const router = inject(Router);

  const authenticated: AuthenticatedResult = oidcSecurityService.authenticated();
  if (!authenticated?.isAuthenticated) {
    return router.parseUrl('/unauthorized');
  }
  return true;
};
