import { ApplicationConfig, inject, provideAppInitializer, provideBrowserGlobalErrorListeners, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { apiInterceptor } from './core/interceptors/http.interceptor';
import { authConfig } from './core/auth/auth.config';
import { AbstractSecurityStorage, authInterceptor, DefaultLocalStorageService, provideAuth } from 'angular-auth-oidc-client';
import { AuthService } from './core/auth/auth.service';
import { tokenInterceptor } from './core/interceptors/token.interceptor';
import { Observable } from 'rxjs';

export function initAuth(authService: AuthService): () => Observable<void> {
  console.log("InitAuth: Starting authentication initialization...");
  return () => authService.initializeAuth();
}


export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideHttpClient(withFetch(), withInterceptors([apiInterceptor, authInterceptor(), tokenInterceptor])),
    provideRouter(routes),
    provideAuth(authConfig),
    // Local storage for tokens
    {
      provide: AbstractSecurityStorage,
      useClass: DefaultLocalStorageService,
    },
    provideAppInitializer(() => {
      const initilizerFn = initAuth(inject(AuthService));
      return initilizerFn();
    }),
  ]
};
