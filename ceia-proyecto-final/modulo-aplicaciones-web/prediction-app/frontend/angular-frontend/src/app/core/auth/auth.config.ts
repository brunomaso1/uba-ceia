import { LogLevel, PassedInitialConfig } from 'angular-auth-oidc-client';

export const authConfig: PassedInitialConfig = {
  config: {
    // https://www.keycloak.org/securing-apps/oidc-layers
    authority: 'http://localhost:7000/realms/prediction-app/',
    redirectUrl: window.location.origin,
    postLogoutRedirectUri: window.location.origin,
    clientId: 'prediction-app-frontend',
    scope: 'openid profile offline_access', // 'openid profile offline_access ' + your scopes
    responseType: 'code',
    silentRenew: true,
    useRefreshToken: true,
    renewTimeBeforeTokenExpiresInSeconds: 30,
    // logLevel: LogLevel.Debug
    autoUserInfo: true,
    // TODO: Fix this
    // forbiddenRoute: '/forbidden',
    // unauthorizedRoute: '/unauthorized',
    // secureRoutes: ['http://localhost:4200/'],
  }
}
