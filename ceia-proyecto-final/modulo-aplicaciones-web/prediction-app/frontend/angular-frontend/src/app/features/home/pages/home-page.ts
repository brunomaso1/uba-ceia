import { Component } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-home-page',
  imports: [RouterLink, MatCardModule, MatButtonModule],
  templateUrl: './home-page.html',
  styleUrl: './home-page.scss'
})
export default class HomePage {

}
