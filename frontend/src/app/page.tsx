'use client'

import { useState } from 'react'
import { PredictionForm } from '@/components/PredictionForm'
import { ResultCard } from '@/components/ResultCard'
import { PredictionResult } from '@/lib/types'
import { Activity, Shield, TrendingUp } from 'lucide-react'

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      {/* Hero Section */}
      <div className="max-w-4xl mx-auto text-center mb-12">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-teal-500/10 border border-teal-500/20 mb-6">
          <Activity className="w-4 h-4 text-teal-400" />
          <span className="text-sm text-teal-400 font-medium">AI-Powered Predictions</span>
        </div>
        
        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold gradient-text mb-4 tracking-tight">
          Health Insurance<br />Cost Predictor
        </h1>
        
        <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed">
          Get accurate estimates of your health insurance costs using advanced machine learning.
          Enter your details below to discover your predicted premium.
        </p>
      </div>

      {/* Features */}
      <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-4 mb-12">
        <div className="flex items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/10">
          <div className="p-2 rounded-lg bg-teal-500/20">
            <TrendingUp className="w-5 h-5 text-teal-400" />
          </div>
          <div>
            <h3 className="font-medium text-white">95% Accuracy</h3>
            <p className="text-sm text-slate-400">ML-powered predictions</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/10">
          <div className="p-2 rounded-lg bg-violet-500/20">
            <Shield className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <h3 className="font-medium text-white">Private & Secure</h3>
            <p className="text-sm text-slate-400">Data never stored</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3 p-4 rounded-xl bg-white/5 border border-white/10">
          <div className="p-2 rounded-lg bg-amber-500/20">
            <Activity className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <h3 className="font-medium text-white">Risk Analysis</h3>
            <p className="text-sm text-slate-400">Personalized insights</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto">
        <div className="glass-card p-8 animate-pulse-glow">
          <h2 className="text-2xl font-semibold text-white mb-6">
            Enter Your Information
          </h2>
          
          <PredictionForm 
            onResult={setResult} 
            isLoading={isLoading}
            setIsLoading={setIsLoading}
          />
        </div>

        {result && (
          <div className="mt-8">
            <ResultCard result={result} />
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="max-w-4xl mx-auto mt-16 text-center">
        <p className="text-sm text-slate-500">
          This tool provides estimates only. Actual insurance costs may vary.
          <br />
          Consult with insurance providers for accurate quotes.
        </p>
      </footer>
    </div>
  )
}

