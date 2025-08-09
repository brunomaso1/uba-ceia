import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ViewPredictionPage } from './view-prediction-page';

describe('ViewPredictionPage', () => {
  let component: ViewPredictionPage;
  let fixture: ComponentFixture<ViewPredictionPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ViewPredictionPage]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ViewPredictionPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
