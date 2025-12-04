import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: 'Health Insurance Cost Predictor',
  description: 'Predict your health insurance costs using advanced machine learning',
  keywords: ['health insurance', 'cost prediction', 'machine learning', 'healthcare'],
  authors: [{ name: 'Insurance Predictor Team' }],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-body antialiased">
        <main className="relative z-10 min-h-screen">
          {children}
        </main>
      </body>
    </html>
  )
}

