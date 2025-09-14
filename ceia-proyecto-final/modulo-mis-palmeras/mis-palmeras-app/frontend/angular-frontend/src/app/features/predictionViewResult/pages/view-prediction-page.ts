import { Component, inject, OnInit } from '@angular/core';
import { MapComponent } from '../components/map-component/map-component';
import { ButtonsBarComponent } from "../components/buttons-bar-component/buttons-bar-component";
import { PredictionResultService } from '../../../shared/services/prediction-result.service';
import { FeatureCollection } from 'geojson';
import { saveAs } from 'file-saver';
import { DownloadService } from '../../predictionDownloadData/services/download.service';
import { firstValueFrom } from 'rxjs';

@Component({
  selector: 'app-view-prediction-page',
  imports: [MapComponent, ButtonsBarComponent],
  templateUrl: './view-prediction-page.html',
  styleUrl: './view-prediction-page.scss'
})
export class ViewPredictionPage {
  predictionResultService = inject(PredictionResultService);
  downloadService = inject(DownloadService);

  geoJSONData: FeatureCollection | undefined = this.predictionResultService.geoJSONData;
  imageId: number | undefined = this.predictionResultService.id;

  async onExportKMLClicked(): Promise<void> {
    if (!this.imageId) {
      alert('No hay predicción disponible para exportar.');
      return;
    }
    const kml: string | null = await firstValueFrom(this.downloadService.downloadKml(this.imageId));
    if (!kml) {
      alert('No hay KML disponible para exportar.');
      return;
    } else {
      try {
        const blob = new Blob([kml || ''], { type: 'application/vnd.google-earth.kml+xml' });
        saveAs(blob, 'prediction.kml');
      } catch (error) {
        const blob = new Blob([kml], { type: 'application/vnd.google-earth.kml+xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction.kml';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    }
  }

  async onDownloadImageClicked(): Promise<void> {
    if (!this.imageId) {
      alert('No hay predicción disponible para exportar.');
      return;
    }
    const imageBlob: Blob | null = await firstValueFrom(this.downloadService.downloadImage(this.imageId));
    if (!imageBlob) {
      alert('No hay imagen disponible para descargar.');
      return;
    } else {
      try {
        const blob = new Blob([imageBlob], { type: 'image/jpeg' });
        saveAs(blob, 'prediction_image.jpg');
      } catch (error) {
        const url = URL.createObjectURL(imageBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction_image.jpg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    }
  }

  async onExportZipClicked(): Promise<void> {
    if (!this.imageId) {
      alert('No hay predicción disponible para exportar.');
      return;
    }
    const zipFileBlob: Blob | null = await firstValueFrom(this.downloadService.downloadZip(this.imageId));
    if (!zipFileBlob) {
      alert('No hay archivo ZIP disponible para descargar.');
      return;
    } else {
      try {
        const blob = new Blob([zipFileBlob], { type: 'application/zip' });
        saveAs(blob, 'prediction.zip');
      } catch (error) {
        const url = URL.createObjectURL(zipFileBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction.zip';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    }
  }
}
