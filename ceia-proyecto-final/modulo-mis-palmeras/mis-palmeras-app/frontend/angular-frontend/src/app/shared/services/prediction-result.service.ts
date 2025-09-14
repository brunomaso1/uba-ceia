import { Injectable } from '@angular/core';
import { FeatureCollection } from 'geojson';

@Injectable({
  providedIn: 'root'
})
export class PredictionResultService {
  id: number | undefined;
  geoJSONData: FeatureCollection | undefined;

  setResult(geoJSONData: FeatureCollection, id: number): void {
    this.id = id;
    this.geoJSONData = geoJSONData;
  }

  clearResult(): void {
    this.id = undefined;
    this.geoJSONData = undefined;
  }
}