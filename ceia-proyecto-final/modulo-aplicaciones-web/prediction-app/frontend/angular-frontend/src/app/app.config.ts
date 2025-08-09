import { ApplicationConfig, inject, provideAppInitializer, provideBrowserGlobalErrorListeners, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';
import { apiInterceptor } from './core/interceptors/http.interceptor';
import { authConfig } from './core/auth/auth.config';
import { authInterceptor, provideAuth } from 'angular-auth-oidc-client';
import { AuthService } from './core/auth/auth.service';
import { tokenInterceptor } from './core/interceptors/token.interceptor';

export function initAuth(authService: AuthService): () => Promise<boolean> {
  return () => authService.init();
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideHttpClient(withFetch(), withInterceptors([apiInterceptor, authInterceptor(), tokenInterceptor])),
    // provideHttpClient(withInterceptors([apiInterceptor, authInterceptor(), tokenInterceptor])),
    provideRouter(routes),
    provideAuth(authConfig),
    provideAppInitializer(() => {
      const initilizerFn = initAuth(inject(AuthService));
      return initilizerFn();
    }),
  ]
};
