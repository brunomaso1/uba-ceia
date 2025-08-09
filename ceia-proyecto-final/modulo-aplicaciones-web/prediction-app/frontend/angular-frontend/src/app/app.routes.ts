import { Routes } from '@angular/router';
import HomePage from './features/home/pages/home-page';
import PredictionPage from './features/predictionUploadData/pages/prediction-page';
import { ViewPredictionPage } from './features/predictionViewResult/pages/view-prediction-page';
import { UnauthorizedPage } from './core/auth/pages/unauthorized-page/unauthorized-page';
import { isAuthenticatedGuard } from './core/auth/isAuthenticated-guard';

export const routes: Routes = [
    {
        path: '',
        component: HomePage,
        title: 'Home page',
        canActivate: [isAuthenticatedGuard]
    },
    {
        path: 'prediction',
        component: PredictionPage,
        title: 'Prediction page',
        canActivate: [isAuthenticatedGuard]
    },
    {
        path: 'viewPrediction',
        component: ViewPredictionPage,
        title: 'View Prediction page',
        canActivate: [isAuthenticatedGuard]
    },
    {
        path: 'unauthorized',
        component: UnauthorizedPage,
        title: 'Unauthorized page',
    }
];
