import { Routes } from '@angular/router';
import HomePage from './features/home/pages/home-page';
import PredictionPage from './features/predictionUploadData/pages/prediction-page';
import { ViewPredictionPage } from './features/predictionViewResult/pages/view-prediction-page';
import { UnauthorizedPage } from './core/auth/pages/unauthorized-page/unauthorized-page';

export const routes: Routes = [
    {
        path: '',
        component: HomePage,
    },
    {
        path: 'prediction',
        component: PredictionPage,
    },
    {
        path: 'viewPrediction',
        component: ViewPredictionPage,
    },
    {
        path: 'unauthorized',
        component: UnauthorizedPage,
    },
    {
        path: '**',
        redirectTo: '',
        pathMatch: 'full'
    }
];
