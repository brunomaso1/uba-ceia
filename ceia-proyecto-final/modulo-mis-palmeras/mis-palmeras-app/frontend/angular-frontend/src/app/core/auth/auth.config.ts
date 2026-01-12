import { LogLevel, PassedInitialConfig } from 'angular-auth-oidc-client';
import { environment } from '../../../environments/environment';

export const authConfig: PassedInitialConfig = {
  config: {
    // https://www.keycloak.org/securing-apps/oidc-layers
    authority: environment.keycloak.authority,
    postLoginRoute: '/',
    redirectUrl: environment.keycloak.redirectUrl,
    postLogoutRedirectUri: environment.keycloak.postLogoutRedirectUri,
    clientId: environment.keycloak.clientId,
    scope: environment.keycloak.scope,
    responseType: 'code',
    silentRenew: true,
    useRefreshToken: true,
    renewTimeBeforeTokenExpiresInSeconds: 30,
    logLevel: environment.production ? LogLevel.Error : LogLevel.Debug,
    autoUserInfo: true,
    forbiddenRoute: '/forbidden',
    unauthorizedRoute: '/unauthorized',
    // secureRoutes: ['http://localhost:4200/'],
  }
};