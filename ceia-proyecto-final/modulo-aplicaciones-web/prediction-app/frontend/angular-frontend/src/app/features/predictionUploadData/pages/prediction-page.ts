import { Component, inject, signal, Signal, viewChild, WritableSignal } from '@angular/core';
import { PredictionButtonsBarComponent } from "../components/prediction-buttons-bar-component/prediction-buttons-bar-component";
import { UploadImageComponent } from "../components/upload-image-component/upload-image-component";
import { UploadJGWComponent } from "../components/upload-jgw-component/upload-jgw-component";
import { JGWModel } from '../models/jgw.model';
import { UploadService } from '../services/upload.service';
import { PredictionResultService } from '../../../shared/services/prediction-result.service';
import { Router } from '@angular/router';
import { FeatureCollection } from 'geojson';
import JSZip from 'jszip';
import { firstValueFrom } from 'rxjs';
import { UploadPredictionDataResponseType } from '../types/upload-prediction-data-response.type';
import { DownloadService } from '../../predictionDownloadData/services/download.service';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

@Component({
  selector: 'app-prediction-page',
  imports: [PredictionButtonsBarComponent, UploadImageComponent, UploadJGWComponent, MatProgressSpinnerModule],
  templateUrl: './prediction-page.html',
  styleUrl: './prediction-page.scss'
})
export default class PredictionPage {
  uploadService: UploadService = inject(UploadService);
  downloadService: DownloadService = inject(DownloadService);

  predictionsResultService: PredictionResultService = inject(PredictionResultService);
  router: Router = inject(Router);
  isLoading: WritableSignal<boolean> = signal(false);
  isButtonDisabled: WritableSignal<boolean> = signal(false);

  uploadImageComponent: Signal<UploadImageComponent> = viewChild.required(UploadImageComponent);
  uploadJGWComponent: Signal<UploadJGWComponent> = viewChild.required(UploadJGWComponent);

  async onPredictionButtonClick(): Promise<void> {
    const imageFile: File | null = this.uploadImageComponent().getImage();
    const jgwModel: JGWModel | null = this.uploadJGWComponent().getJGWModel();

    if (!(imageFile && jgwModel)) {
      alert('Por favor, asegúrese de que se ha seleccionado una imagen y un archivo JGW antes de realizar la predicción.');
      return;
    }

    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('jgw', JSON.stringify(jgwModel));

    try {
      this.isLoading.set(true);
      this.isButtonDisabled.set(true);
      const response: UploadPredictionDataResponseType = await firstValueFrom(this.uploadService.uploadPredictionData(formData));
      if (response.status === 'completed') {
        const geoJSONData: FeatureCollection = await firstValueFrom(this.downloadService.downloadGeoJson(response.id));
        this.predictionsResultService.setResult(geoJSONData, response.id);
        this.isLoading.set(false);
        this.router.navigate(['/viewPrediction']);
      }
    } catch (error) {
      console.error('Error al realizar la predicción:', error);
      alert('Error al realizar la predicción. Por favor, inténtelo de nuevo más tarde.');
    }
  }
}