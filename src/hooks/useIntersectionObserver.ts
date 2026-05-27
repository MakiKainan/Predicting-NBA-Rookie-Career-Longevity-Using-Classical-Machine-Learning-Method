import { useEffect, useRef, useState } from 'react';

export function useIntersectionObserver<T extends HTMLElement = HTMLElement>(
  options: IntersectionObserverInit = { threshold: 0.1 }
) {
  const [isVisible, setIsVisible] = useState(true);
  const ref = useRef<T>(null);

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return;

    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setIsVisible(true);
      }
    }, options);
    
    const currentRef = ref.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.disconnect();
      }
    };
  }, [options.threshold]); // Simplified dependency

  return [ref, isVisible] as const;
}
