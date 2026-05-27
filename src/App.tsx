/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import AboutSection from './components/AboutSection';
import PredictSection from './components/PredictSection';
import ModelSection from './components/ModelSection';

export default function App() {
  return (
    <div className="relative min-h-screen w-full flex flex-col bg-background selection:bg-primary selection:text-white">
      <Navbar />
      <HeroSection />
      <AboutSection />
      <PredictSection />
      <ModelSection />
    </div>
  );
}
