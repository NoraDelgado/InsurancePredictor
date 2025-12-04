'use client'

import { motion } from 'framer-motion'
import { DollarSign, Activity, Shield } from 'lucide-react'
import { PredictionResult } from '@/lib/types'

interface ResultCardProps {
  result: PredictionResult
}

export function ResultCard({ result }: ResultCardProps) {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  // Determine cost tier for visual feedback
  const getCostTier = (cost: number) => {
    if (cost < 5000) return { label: 'Low', color: 'text-emerald-400', bg: 'bg-emerald-500/20' }
    if (cost < 15000) return { label: 'Moderate', color: 'text-amber-400', bg: 'bg-amber-500/20' }
    if (cost < 30000) return { label: 'High', color: 'text-orange-400', bg: 'bg-orange-500/20' }
    return { label: 'Very High', color: 'text-red-400', bg: 'bg-red-500/20' }
  }

  const costTier = getCostTier(result.predicted_charge)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className="glass-card p-8"
    >
      {/* Main Prediction */}
      <div className="text-center mb-8">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-teal-500/20 to-violet-500/20 mb-4"
        >
          <DollarSign className="w-8 h-8 text-teal-400" />
        </motion.div>
        
        <h2 className="text-xl font-medium text-slate-400 mb-3">
          Estimated Annual Insurance Cost
        </h2>
        
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, type: 'spring', stiffness: 150 }}
          className="text-5xl md:text-6xl font-bold gradient-text-primary mb-4"
        >
          {formatCurrency(result.predicted_charge)}
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${costTier.bg}`}
        >
          <Activity className={`w-4 h-4 ${costTier.color}`} />
          <span className={`font-semibold ${costTier.color}`}>{costTier.label} Cost Range</span>
        </motion.div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.5 }}
          className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50"
        >
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-teal-500/20">
              <Shield className="w-5 h-5 text-teal-400" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Monthly Estimate</p>
              <p className="text-lg font-semibold text-white">
                {formatCurrency(result.predicted_charge / 12)}
              </p>
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.6 }}
          className="p-4 rounded-xl bg-slate-800/50 border border-slate-700/50"
        >
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-violet-500/20">
              <Activity className="w-5 h-5 text-violet-400" />
            </div>
            <div>
              <p className="text-sm text-slate-400">Model Version</p>
              <p className="text-lg font-semibold text-white font-mono">
                v{result.model_version}
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Tips Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="p-6 rounded-xl bg-gradient-to-br from-teal-500/10 to-violet-500/10 border border-teal-500/20"
      >
        <h4 className="font-semibold text-white mb-3 flex items-center gap-2">
          <span className="text-teal-400">💡</span> Tips to Lower Your Costs
        </h4>
        <ul className="space-y-2 text-slate-300 text-sm">
          <li className="flex items-start gap-2">
            <span className="text-teal-400 mt-1">•</span>
            <span>Quitting smoking can significantly reduce your insurance premium</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-teal-400 mt-1">•</span>
            <span>Maintaining a healthy BMI through diet and exercise helps</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-teal-400 mt-1">•</span>
            <span>Regular health check-ups can help manage blood pressure</span>
          </li>
        </ul>
      </motion.div>

      {/* Disclaimer */}
      <p className="mt-6 text-xs text-center text-slate-500">
        This prediction is for informational purposes only and should not be considered as financial or medical advice.
      </p>
    </motion.div>
  )
}
