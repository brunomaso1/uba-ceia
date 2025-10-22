import { LogLevel, PassedInitialConfig } from 'angular-auth-oidc-client';

export const authConfig: PassedInitialConfig = {
  config: {
    // https://www.keycloak.org/securing-apps/oidc-layers
    authority: 'http://localhost:7000/realms/mis-palmeras-app/',
    postLoginRoute: '/',
    redirectUrl: window.location.origin,
    postLogoutRedirectUri: window.location.origin,
    clientId: 'prediction-app-frontend',
    scope: 'openid profile email offline_access',
    responseType: 'code',
    silentRenew: true,
    useRefreshToken: true,
    renewTimeBeforeTokenExpiresInSeconds: 30,
    logLevel: LogLevel.Debug,
    autoUserInfo: true,
    forbiddenRoute: '/forbidden',
    unauthorizedRoute: '/unauthorized',
    // secureRoutes: ['http://localhost:4200/'],
  }
}
