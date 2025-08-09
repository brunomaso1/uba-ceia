import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { FeatureCollection } from 'geojson';
import { Observable, of } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class DownloadService {
  private readonly downloadGeoJsonApi = '/geolocalization/downloadGeoJSON/{image_id}';
  private readonly downloadImageApi = '/image/downloadAnnotated/{image_id}';
  private readonly downloadKmlApi = '/geolocalization/downloadKML/{image_id}';
  private readonly downloadZipApi = '/zip/download/{image_id}';
  private http: HttpClient = inject(HttpClient);

  downloadGeoJson(imageId: number): Observable<FeatureCollection> {
    const url = this.downloadGeoJsonApi.replace('{image_id}', imageId.toString());
    return this.http.get<FeatureCollection>(url);
  }

  downloadImage(imageId: number): Observable<Blob | null> {
    const url = this.downloadImageApi.replace('{image_id}', imageId.toString());
    return this.http.get(url, { responseType: 'blob' });
  }

  downloadKml(imageId: number): Observable<string | null> {
    const url = this.downloadKmlApi.replace('{image_id}', imageId.toString());
    return this.http.get(url, { responseType: 'text' });
  }

  downloadZip(imageId: number): Observable<Blob | null> {
    const url = this.downloadZipApi.replace('{image_id}', imageId.toString());
    return this.http.get(url, { responseType: 'blob' });
  }
}
