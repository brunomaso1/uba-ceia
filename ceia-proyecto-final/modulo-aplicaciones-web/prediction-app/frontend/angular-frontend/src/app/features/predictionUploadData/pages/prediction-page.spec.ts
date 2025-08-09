import { ComponentFixture, TestBed } from '@angular/core/testing';
import PredictionPage from './prediction-page';

describe('PredictionPage', () => {
  let component: PredictionPage;
  let fixture: ComponentFixture<PredictionPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PredictionPage]
    })
      .compileComponents();

    fixture = TestBed.createComponent(PredictionPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
