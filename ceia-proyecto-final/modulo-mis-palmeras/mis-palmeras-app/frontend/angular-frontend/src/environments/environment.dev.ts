// For production have to do: ng build --prod

export const environment = {
    production: true,
    keycloak: {
        authority: 'https://auth.picudo-rojo-desarrollo.org/realms/mis-palmeras-app',
        redirectUrl: 'https://app.picudo-rojo-desarrollo.org',
        postLogoutRedirectUri: 'https://app.picudo-rojo-desarrollo.org',
        clientId: 'prediction-app-frontend',
        scope: 'openid profile email offline_access',
    },
    backend: {
        apiBaseUrl: 'http://api.picudo-rojo-desarrollo.org/apiv1',
    }
};