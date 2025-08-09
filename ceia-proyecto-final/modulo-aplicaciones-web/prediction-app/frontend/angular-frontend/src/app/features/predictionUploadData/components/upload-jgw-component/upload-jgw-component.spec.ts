import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadJGWComponent } from './upload-jgw-component';

describe('UploadJGWComponent', () => {
  let component: UploadJGWComponent;
  let fixture: ComponentFixture<UploadJGWComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [UploadJGWComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(UploadJGWComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
