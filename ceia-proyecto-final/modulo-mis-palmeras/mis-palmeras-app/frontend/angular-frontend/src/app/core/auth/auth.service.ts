import { Injectable, inject, signal } from '@angular/core';
import { BehaviorSubject, firstValueFrom, Observable, of, throwError } from 'rxjs';
import { LoginResponse, OidcSecurityService, UserDataResult } from 'angular-auth-oidc-client';
import { map, catchError, take, tap, filter, switchAll, switchMap } from 'rxjs/operators';
import { Router } from '@angular/router';
import { User } from './models/user.model';

@Injectable({
    providedIn: 'root'
})
export class AuthService {
    private readonly oidcSecurityService = inject(OidcSecurityService);
    private readonly router = inject(Router);
    private readonly currentUser = new BehaviorSubject<User | null>(null);

    onInit(): void {
        this.oidcSecurityService.userData$.subscribe((userData: UserDataResult) => {
            if (userData && userData.userData) {
                console.log("Datos de usuario obtenidos:", userData.userData);
                const user: User = {
                    name: userData.userData.name || userData.userData.preferred_username,
                    email: userData.userData.email,
                    sub: userData.userData.sub,
                };
                this.currentUser.next(user);
            } else {
                console.log("No se encontraron datos de usuario.");
                this.currentUser.next(null);
            }
        });
    }

    initializeAuth(): Observable<void> {
        return this.oidcSecurityService.checkAuth().pipe(
            take(1),
            switchMap(({ isAuthenticated }) => {
                if (isAuthenticated) {
                    return of(void 0);
                } else {
                    this.oidcSecurityService.authorize();
                    return this.oidcSecurityService.stsCallback$;
                }
            })
        );
    }

    logout(): void {
        this.oidcSecurityService.logoff().subscribe((result) => {
            this.currentUser.next(null);
            this.router.navigate(['/']);
        });
    }

    getCurrentUser(): Observable<User | null> {
        return this.currentUser.asObservable();
    }

    isAuthenticated(): Observable<boolean> {
        return this.oidcSecurityService.isAuthenticated$.pipe(
            tap(isAuth => console.log("Estado de autenticación:", isAuth)),
            map(result => result.isAuthenticated));
    }

    getToken(): Observable<string> {
        return this.oidcSecurityService.getIdToken();
    }

    onDestroy() {
        this.currentUser.complete();
    }
}