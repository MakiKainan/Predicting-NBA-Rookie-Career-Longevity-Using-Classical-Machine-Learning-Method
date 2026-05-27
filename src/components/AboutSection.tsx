import { useIntersectionObserver } from '../hooks/useIntersectionObserver';

export default function AboutSection() {
  const [ref, isVisible] = useIntersectionObserver();

  return (
    <section id="about" ref={ref} className="py-24 px-6 max-w-7xl mx-auto w-full">
      <div className="flex flex-col md:flex-row gap-12 items-center">
        <div className={`flex-1 ${isVisible ? 'animate-fade-rise' : 'opacity-0'}`}>
          <h2 className="text-5xl md:text-6xl font-display text-secondary uppercase leading-none mb-6">About Us</h2>
          <p className="text-muted-foreground text-lg leading-relaxed mb-6 font-body">
            We are dedicated to bringing cutting-edge machine learning technology to basketball analytics.
            Our models analyze thousands of data points to deliver accurate predictions and deep insights into the game.
          </p>
          <div className="flex gap-4">
            <div className="w-16 h-2 bg-primary"></div>
            <div className="w-16 h-2 bg-secondary"></div>
          </div>
        </div>
        <div className={`flex-1 w-full bg-muted min-h-[300px] flex items-center justify-center p-8 border-4 border-secondary relative ${isVisible ? 'animate-fade-rise-delay' : 'opacity-0'}`}>
           <div className="absolute top-4 left-4 w-12 h-12 border-t-4 border-l-4 border-primary"></div>
           <div className="absolute bottom-4 right-4 w-12 h-12 border-b-4 border-r-4 border-primary"></div>
           <span className="text-2xl font-display text-muted-foreground uppercase tracking-widest text-center">Analytics Dashboard</span>
        </div>
      </div>
    </section>
  );
}
