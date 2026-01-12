// For production have to do: ng build --prod

export const environment = {
    production: true,
    keycloak: {
        authority: 'https://auth.picudo-rojo.org/realms/mis-palmeras-app',
        redirectUrl: 'https://app.picudo-rojo.org',
        postLogoutRedirectUri: 'https://app.picudo-rojo.org',
        clientId: 'prediction-app-frontend',
        scope: 'openid profile email offline_access',
    },
    backend: {
        apiBaseUrl: 'https://api.picudo-rojo.org/apiv1',
    }
};