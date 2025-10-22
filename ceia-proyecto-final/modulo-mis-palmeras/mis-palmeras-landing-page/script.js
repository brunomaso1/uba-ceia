// Mis Palmeras Landing Page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeMobileMenu();
    initializeSmoothScrolling();
    initializeHeaderScrollEffect();
    initializeAnimationsOnScroll();
    initializeFormValidation();
    initializeLoadingStates();
});

// Mobile Menu Toggle
function initializeMobileMenu() {
    const mobileMenuToggle = document.querySelector('[data-landingsite-mobile-menu-toggle]');
    const mobileMenu = document.querySelector('[data-landingsite-mobile-menu]');
    
    if (mobileMenuToggle && mobileMenu) {
        mobileMenuToggle.addEventListener('click', function() {
            const isHidden = mobileMenu.classList.contains('hidden');
            
            if (isHidden) {
                mobileMenu.classList.remove('hidden');
                mobileMenu.classList.add('mobile-menu-visible');
                mobileMenuToggle.setAttribute('aria-expanded', 'true');
                // Change icon to close
                const icon = mobileMenuToggle.querySelector('i');
                if (icon) {
                    icon.classList.remove('fa-bars');
                    icon.classList.add('fa-times');
                }
            } else {
                mobileMenu.classList.add('hidden');
                mobileMenu.classList.remove('mobile-menu-visible');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                // Change icon to hamburger
                const icon = mobileMenuToggle.querySelector('i');
                if (icon) {
                    icon.classList.remove('fa-times');
                    icon.classList.add('fa-bars');
                }
            }
        });

        // Close mobile menu when clicking on a link
        const mobileMenuLinks = mobileMenu.querySelectorAll('a');
        mobileMenuLinks.forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.add('hidden');
                mobileMenu.classList.remove('mobile-menu-visible');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                const icon = mobileMenuToggle.querySelector('i');
                if (icon) {
                    icon.classList.remove('fa-times');
                    icon.classList.add('fa-bars');
                }
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!mobileMenuToggle.contains(event.target) && !mobileMenu.contains(event.target)) {
                mobileMenu.classList.add('hidden');
                mobileMenu.classList.remove('mobile-menu-visible');
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                const icon = mobileMenuToggle.querySelector('i');
                if (icon) {
                    icon.classList.remove('fa-times');
                    icon.classList.add('fa-bars');
                }
            }
        });
    }
}

// Smooth Scrolling for anchor links
function initializeSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Header scroll effect
function initializeHeaderScrollEffect() {
    const header = document.getElementById('global-header');
    if (header) {
        let lastScrollY = window.scrollY;
        
        window.addEventListener('scroll', function() {
            const currentScrollY = window.scrollY;
            
            if (currentScrollY > 100) {
                header.classList.add('sticky-header');
            } else {
                header.classList.remove('sticky-header');
            }
            
            // Hide/show header on scroll
            if (currentScrollY > lastScrollY && currentScrollY > 200) {
                header.style.transform = 'translateY(-100%)';
            } else {
                header.style.transform = 'translateY(0)';
            }
            
            lastScrollY = currentScrollY;
        });
    }
}

// Animations on scroll
function initializeAnimationsOnScroll() {
    const animateElements = document.querySelectorAll('.code-section');
    
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    animateElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'all 0.6s ease-out';
        observer.observe(element);
    });
}

// Form validation (if there are forms)
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            const data = Object.fromEntries(formData);
            
            // Add loading state
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.classList.add('loading');
                submitButton.textContent = 'Enviando...';
                submitButton.disabled = true;
            }
            
            // Simulate form submission
            setTimeout(() => {
                if (submitButton) {
                    submitButton.classList.remove('loading');
                    submitButton.textContent = 'Formulario Enviado';
                    submitButton.style.backgroundColor = 'var(--accent-color)';
                }
                
                // Reset form after 3 seconds
                setTimeout(() => {
                    form.reset();
                    if (submitButton) {
                        submitButton.textContent = 'Enviar';
                        submitButton.disabled = false;
                        submitButton.style.backgroundColor = '';
                    }
                }, 3000);
            }, 2000);
        });
    });
}

// Loading states for buttons and links
function initializeLoadingStates() {
    const buttons = document.querySelectorAll('a[href*="contactar"], a[href*="demo"]');
    
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            if (this.href.includes('#') || this.href.includes('javascript:')) {
                return; // Skip for anchor links
            }
            
            // Add loading state
            const originalText = this.textContent;
            this.textContent = 'Cargando...';
            this.style.pointerEvents = 'none';
            
            // Reset after navigation (or timeout)
            setTimeout(() => {
                this.textContent = originalText;
                this.style.pointerEvents = '';
            }, 3000);
        });
    });
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Add some keyboard navigation support
document.addEventListener('keydown', function(e) {
    // Close mobile menu with Escape key
    if (e.key === 'Escape') {
        const mobileMenu = document.querySelector('[data-landingsite-mobile-menu]');
        const mobileMenuToggle = document.querySelector('[data-landingsite-mobile-menu-toggle]');
        
        if (mobileMenu && !mobileMenu.classList.contains('hidden')) {
            mobileMenu.classList.add('hidden');
            mobileMenu.classList.remove('mobile-menu-visible');
            if (mobileMenuToggle) {
                mobileMenuToggle.setAttribute('aria-expanded', 'false');
                const icon = mobileMenuToggle.querySelector('i');
                if (icon) {
                    icon.classList.remove('fa-times');
                    icon.classList.add('fa-bars');
                }
            }
        }
    }
});

// Performance optimization: Lazy load images
function initializeLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
}

// Initialize lazy loading if needed
initializeLazyLoading();