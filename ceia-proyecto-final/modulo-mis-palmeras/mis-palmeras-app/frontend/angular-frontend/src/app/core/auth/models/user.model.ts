export interface User {
    name: string;
    email: string;
    sub?: string;
    isNominated?: boolean;
    // otros campos que vengan en el token
}
