const SUBTEXT = "Empowering teams and fans with state-of-the-art predictive models. Step into the future of sports analytics where data meets the hardwood.";
const CTA_LABEL = "View Predictions";

export default function HeroSection() {
  return (
    <section id="home" className="relative z-10 w-full flex flex-col items-center justify-center text-center px-6 py-40 md:py-48 bg-slate-950 overflow-hidden">
      {/* Background Video */}
      <video 
        src="/Lebron.mp4" 
        autoPlay 
        loop 
        muted 
        playsInline 
        className="absolute inset-0 w-full h-full object-cover z-0 opacity-45 pointer-events-none"
      />

      {/* Sleek dark gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-b from-slate-950/80 via-slate-950/50 to-slate-950/80 z-1 pointer-events-none"></div>
      
      {/* Smooth transition to the next white section */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-white via-white/40 to-transparent z-2 pointer-events-none"></div>
      
      <div className="relative z-10 max-w-5xl mx-auto flex flex-col items-center">
        {/* Analytics Badge */}
        <div className="inline-block mb-10 px-4 py-1.5 bg-white/10 text-white font-bold uppercase tracking-widest text-xs border-l-4 border-primary animate-fade-rise font-body backdrop-blur-sm">
           Next Generation Analytics
        </div>
        
        {/* Title */}
        <h1 className="text-6xl sm:text-7xl md:text-[9rem] font-display uppercase leading-[0.85] animate-fade-rise text-white flex flex-col items-center justify-center">
          <span className="block italic tracking-tight relative">
            Machine
            <span className="absolute -bottom-2 md:-bottom-4 left-0 w-full h-2 md:h-4 bg-primary"></span>
          </span>
          <span className="block text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-300 drop-shadow-md mt-4 md:mt-8 tracking-tight">Learning</span>
        </h1>
        
        {/* Subtext */}
        <p className="animate-fade-rise-delay text-slate-300 text-lg sm:text-xl max-w-2xl mt-14 leading-relaxed font-body">
          {SUBTEXT}
        </p>
        
        {/* CTAs */}
        <div className="flex flex-col sm:flex-row gap-6 mt-14 animate-fade-rise-delay-2 w-full sm:w-auto">
          <a href="#predict" className="sporty-btn inline-flex items-center justify-center px-12 py-5 text-lg w-full sm:w-auto">
            {CTA_LABEL}
          </a>
          <a href="#model" className="bg-transparent text-white border-[3px] border-white inline-flex items-center justify-center px-12 py-5 text-lg font-bold uppercase tracking-widest hover:bg-white/15 transition-all duration-300 font-display w-full sm:w-auto">
            Our Models
          </a>
        </div>
      </div>
    </section>
  );
}
