import { inject, Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, switchMap } from 'rxjs';
import { UploadPredictionDataResponseType } from '../types/upload-prediction-data-response.type';

@Injectable({
  providedIn: 'root'
})
export class UploadService {
  private readonly imageUploadApi = '/image/upload';
  private readonly jgwUploadApi = '/jgw/upload/{image_id}';
  private readonly syncPredictionsApi = '/predictions/generate_sync_prediction/{image_id}';
  private http: HttpClient = inject(HttpClient);

  uploadPredictionData(formData: FormData): Observable<any> {
    const imageFile: File | null = formData.get('image') as File | null;
    if (!imageFile) {
      return of({ error: 'No image file provided' });
    }

    const imageFormData = new FormData();
    imageFormData.append('file', imageFile);

    return this.http.post<{ id: number, name: string }>(this.imageUploadApi, imageFormData).pipe(
      switchMap(response => {
        const imageId = response.id;
        const jgwJson: string | null = formData.get('jgw') as string | null;

        const url = this.jgwUploadApi.replace('{image_id}', imageId.toString());
        if (jgwJson) {
          return this.http.post<{ id: number, name: string }>(url, JSON.parse(jgwJson)).pipe(
            switchMap(() => {
              const url = this.syncPredictionsApi.replace('{image_id}', imageId.toString());
              return this.http.post<UploadPredictionDataResponseType>(url, {})
            })
          );
        }
        return of({ error: 'No JGW data provided' });
      })
    );
  }
}