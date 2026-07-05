import { motion, AnimatePresence } from 'framer-motion'

export default function AnomalyAlert({ anomaly, reconError }) {
  return (
    <AnimatePresence mode="wait">
      {anomaly ? (
        <motion.div
          key="danger"
          className="alert danger"
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{
            opacity: 1,
            scale: 1,
            boxShadow: [
              '0 0 18px rgba(255,45,85,0.25), inset 0 0 24px rgba(255,45,85,0.08)',
              '0 0 44px rgba(255,45,85,0.6), inset 0 0 34px rgba(255,45,85,0.16)',
              '0 0 18px rgba(255,45,85,0.25), inset 0 0 24px rgba(255,45,85,0.08)',
            ],
          }}
          exit={{ opacity: 0, scale: 0.97 }}
          transition={{ boxShadow: { duration: 1.1, repeat: Infinity }, default: { duration: 0.35 } }}
        >
          <motion.svg
            width="30" height="30" viewBox="0 0 24 24" fill="none"
            animate={{ scale: [1, 1.18, 1] }}
            transition={{ duration: 1.1, repeat: Infinity }}
          >
            <path d="M12 2L1 21h22L12 2z" stroke="#ff2d55" strokeWidth="1.8" fill="rgba(255,45,85,0.15)" />
            <path d="M12 9v5" stroke="#ff2d55" strokeWidth="2.2" strokeLinecap="round" />
            <circle cx="12" cy="17.4" r="1.2" fill="#ff2d55" />
          </motion.svg>
          <div>
            <div className="alert-title">⚠ ANOMALY DETECTED — HIGH-IMPACT EVENT</div>
            <div className="alert-sub">VAE RECONSTRUCTION ERROR {reconError.toFixed(3)} — EXCEEDS THRESHOLD 0.500</div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          key="ok"
          className="alert ok"
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.98 }}
          transition={{ duration: 0.35 }}
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="9.5" stroke="#3dff8f" strokeWidth="1.6" opacity="0.8" />
            <path d="M8 12.5l2.6 2.6L16 9.5" stroke="#3dff8f" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>
            <div className="alert-title">ALL SYSTEMS NOMINAL</div>
            <div className="alert-sub">VAE RECONSTRUCTION ERROR {reconError.toFixed(3)} — WITHIN TOLERANCE</div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
