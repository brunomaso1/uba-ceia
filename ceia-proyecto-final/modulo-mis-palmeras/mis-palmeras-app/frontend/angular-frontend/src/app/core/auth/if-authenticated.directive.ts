import {
  Directive,
  inject,
  input,
  effect,
  TemplateRef,
  ViewContainerRef,
} from "@angular/core";
import { toSignal } from "@angular/core/rxjs-interop";
import { AuthService } from "./auth.service";

@Directive({
  selector: "[ifAuthenticated]",
})
export class IfAuthenticatedDirective<T> {
  private readonly authService = inject(AuthService);
  private readonly templateRef = inject(TemplateRef<T>);
  private readonly viewContainer = inject(ViewContainerRef);

  readonly condition = input.required<boolean>({ alias: 'ifAuthenticated' });

  private hasView = false;

  constructor() {
    effect(() => {
      const isAuthenticated = toSignal(this.authService.isAuthenticated())();
      const shouldShow = this.condition();

      const authRequired = isAuthenticated && shouldShow;
      const unauthRequired = !isAuthenticated && !shouldShow;

      if ((authRequired || unauthRequired) && !this.hasView) {
        this.viewContainer.createEmbeddedView(this.templateRef);
        this.hasView = true;
      } else if (this.hasView && !(authRequired || unauthRequired)) {
        this.viewContainer.clear();
        this.hasView = false;
      }
    });
  }
}
