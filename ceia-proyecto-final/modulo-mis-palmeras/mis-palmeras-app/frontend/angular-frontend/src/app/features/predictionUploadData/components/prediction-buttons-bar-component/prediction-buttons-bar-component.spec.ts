import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PredictionButtonsBarComponent } from './prediction-buttons-bar-component';

describe('PredictionButtonsBarComponent', () => {
  let component: PredictionButtonsBarComponent;
  let fixture: ComponentFixture<PredictionButtonsBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PredictionButtonsBarComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PredictionButtonsBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
