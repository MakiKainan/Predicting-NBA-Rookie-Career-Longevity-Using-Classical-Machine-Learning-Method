const NAV_LINKS = [
  { label: 'Home', href: '#home', active: true },
  { label: 'About Us', href: '#about', active: false },
  { label: 'Predict', href: '#predict', active: false },
  { label: 'Model', href: '#model', active: false },
];

const CTA_LABEL = "Get Started";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-50 w-full bg-white border-b-4 border-primary shadow-sm">
      <nav className="w-full px-6 py-4 max-w-7xl mx-auto flex justify-between items-center">
        <a href="#home" className="flex items-center">
          <img 
            src="/logo.png" 
            alt="NBA Machine Learning" 
            className="h-12 md:h-16 object-contain"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none';
              document.getElementById('logo-fallback')!.style.display = 'block';
            }}
          />
          <div id="logo-fallback" className="hidden text-3xl font-display text-secondary tracking-tight uppercase italic">
            NBA<span className="text-primary">ML</span>
          </div>
        </a>
        <div className="hidden md:flex gap-8 items-center text-sm font-bold uppercase tracking-widest font-body">
          {NAV_LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              className={`transition-colors cursor-pointer border-b-[3px] py-1 ${
                  link.active ? 'border-primary text-secondary' : 'border-transparent text-muted-foreground hover:text-secondary hover:border-secondary'
                }`}
              >
                {link.label}
              </a>
            ))}
          </div>
          <button className="hidden md:block sporty-btn px-8 py-3 text-sm">
            {CTA_LABEL}
          </button>
          
          {/* Mobile Menu Toggle */}
          <div className="md:hidden text-secondary p-2 cursor-pointer hover:text-primary transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
          </div>
      </nav>
    </header>
  );
}
