import { inject, Injectable } from '@angular/core';
import { OidcSecurityService } from 'angular-auth-oidc-client';

@Injectable({ providedIn: 'root' })
export class AuthService {
    oidcSecurityService: OidcSecurityService = inject(OidcSecurityService);

    init(): Promise<boolean> {
        return new Promise<boolean>((resolve) => {
            this.oidcSecurityService.checkAuth()
                .subscribe(({ isAuthenticated }) => {
                    if (!isAuthenticated) {
                        this.oidcSecurityService.authorize();
                    }
                    resolve(isAuthenticated);
                });
        });
    }
}