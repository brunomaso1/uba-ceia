import { TestBed } from '@angular/core/testing';

import { PredictionResultService } from './prediction-result.service';

describe('PredictionResultService', () => {
  let service: PredictionResultService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PredictionResultService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
