import { AsyncPipe, JsonPipe } from "@angular/common";
import { Component, inject, Signal, WritableSignal } from "@angular/core";
import { MatButtonModule } from "@angular/material/button";
import { MatIconModule } from "@angular/material/icon";
import { MatToolbarModule } from "@angular/material/toolbar";
import { RouterLink, RouterLinkActive } from "@angular/router";
import { OidcSecurityService, UserDataResult } from "angular-auth-oidc-client";

@Component({
  selector: "app-layout-header",
  templateUrl: "./header.component.html",
  styleUrls: ["./header.component.scss"],
  imports: [RouterLink, RouterLinkActive, MatToolbarModule, MatButtonModule, MatIconModule],
})
export class HeaderComponent {

  oidcSecurityService = inject(OidcSecurityService);

  userData: Signal<UserDataResult> = this.oidcSecurityService.userData;

  logout() {
    this.oidcSecurityService.logoff().subscribe({
      next: () => console.log('User logged out successfully'),
      error: (err) => console.error('Error during logout:', err)
    });
  }
}