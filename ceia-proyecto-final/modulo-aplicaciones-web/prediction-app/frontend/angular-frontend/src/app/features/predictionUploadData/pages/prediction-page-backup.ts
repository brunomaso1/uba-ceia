// import { Component, inject, signal, Signal, viewChild, WritableSignal } from '@angular/core';
// import { PredictionButtonsBarComponent } from "../components/prediction-buttons-bar-component/prediction-buttons-bar-component";
// import { UploadImageComponent } from "../components/upload-image-component/upload-image-component";
// import { UploadJGWComponent } from "../components/upload-jgw-component/upload-jgw-component";
// import { JGWModel } from '../models/jgw-model';
// import { UploadService } from '../services/upload-service';
// import { PredictionResultService } from '../../../shared/services/prediction-result-service';
// import { Router } from '@angular/router';
// import { FeatureCollection } from 'geojson';
// import JSZip from 'jszip';
// import { firstValueFrom } from 'rxjs';

// @Component({
//   selector: 'app-prediction-page',
//   imports: [PredictionButtonsBarComponent, UploadImageComponent, UploadJGWComponent],
//   templateUrl: './prediction-page.html',
//   styleUrl: './prediction-page.css'
// })
// export default class PredictionPage {
//   uploadService: UploadService = inject(UploadService);
//   predictionsResultService: PredictionResultService = inject(PredictionResultService);
//   router: Router = inject(Router);
//   isLoading: WritableSignal<boolean> = signal(false);
//   isButtonDisabled: WritableSignal<boolean> = signal(false);

//   uploadImageComponent: Signal<UploadImageComponent> = viewChild.required(UploadImageComponent);
//   uploadJGWComponent: Signal<UploadJGWComponent> = viewChild.required(UploadJGWComponent);

//   async onPredictionButtonClick(): Promise<void> {
//     const imageFile: File | null = this.uploadImageComponent().getImage();
//     const jgwModel: JGWModel | null = this.uploadJGWComponent().getJGWModel();

//     if (!(imageFile && jgwModel)) {
//       alert('Por favor, asegúrese de que se ha seleccionado una imagen y un archivo JGW antes de realizar la predicción.');
//       return;
//     }

//     const formData = new FormData();
//     formData.append('image', imageFile);
//     formData.append('jgw', JSON.stringify(jgwModel));

//     try {
//       this.isLoading.set(true);
//       this.isButtonDisabled.set(true);
//       const response: Blob = await firstValueFrom(this.uploadService.uploadPredictionData(formData));
//       const jszip = new JSZip();
//       const zipContent = await jszip.loadAsync(response);

//       let imageBlob: Blob | null = null;
//       let kmlText: string | null = null;
//       let geoJSONData: FeatureCollection | null = null;

//       const fileNames = Object.keys(zipContent.files);
//       for (const filename of fileNames) {
//         const file = zipContent.files[filename];
//         if (!file.dir) {
//           if (filename.match(/\.(jpg|jpeg|png)$/i)) {
//             const arrayBuffer = await file.async("arraybuffer");
//             imageBlob = new Blob([arrayBuffer], { type: 'image/jpeg' });
//           } else if (filename.endsWith('.kml')) {
//             kmlText = await file.async("string");
//           } else if (filename.endsWith('.geojson') || filename.endsWith('.json')) {
//             const geojsonString = await file.async("string");
//             geoJSONData = JSON.parse(geojsonString) as FeatureCollection;
//           }
//         }
//       }

//       if (!imageBlob || !kmlText || !geoJSONData) {
//         alert('No se han encontrado todos los archivos requeridos en el zip.');
//         return;
//       }

//       this.predictionsResultService.setResult(imageBlob, geoJSONData, kmlText);
//       this.isLoading.set(false);
//       this.router.navigate(['/viewPrediction']);
//     } catch (error) {
//       console.error('Error al realizar la predicción:', error);
//       alert('Error al realizar la predicción. Por favor, inténtelo de nuevo más tarde.');
//     }
//   }
// }