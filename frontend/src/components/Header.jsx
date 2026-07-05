import { motion } from 'framer-motion'

export default function Header({ running, override, step }) {
  const status = override === 'locked' ? 'locked' : running ? 'online' : 'paused'
  const statusText =
    override === 'locked' ? 'INTERSECTION LOCKED' : running ? 'SYSTEM ONLINE' : 'SIMULATION PAUSED'
  const dotClass = override === 'locked' ? 'r' : running ? 'g' : 'a'

  return (
    <motion.header
      className="panel hdr"
      initial={{ y: -24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="hdr-brand">
        <motion.div
          className="hdr-logo"
          animate={{ boxShadow: [
            '0 0 18px rgba(0,229,255,0.25), inset 0 0 12px rgba(0,229,255,0.1)',
            '0 0 30px rgba(0,229,255,0.5), inset 0 0 18px rgba(0,229,255,0.2)',
            '0 0 18px rgba(0,229,255,0.25), inset 0 0 12px rgba(0,229,255,0.1)',
          ] }}
          transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
            <path d="M12 2L3 7v10l9 5 9-5V7l-9-5z" stroke="#00e5ff" strokeWidth="1.4" />
            <circle cx="12" cy="12" r="3.2" fill="#00e5ff" opacity="0.9" />
            <circle cx="12" cy="12" r="5.6" stroke="#ff3df2" strokeWidth="0.9" opacity="0.7" />
          </svg>
        </motion.div>
        <div>
          <div className="hdr-title">NEXUS TRAFFIC CORE</div>
          <div className="hdr-tag">Multimodal AI Optimization Grid</div>
        </div>
      </div>

      <div className="hdr-right">
        <div className="chip">
          STEP <strong>{String(step).padStart(4, '0')}</strong>
        </div>
        <div className="chip">
          MODELS <strong>YOLOv8 · LSTM · VAE · BERT · PPO</strong>
        </div>
        <motion.div
          className={`chip ${status}`}
          animate={{ opacity: running || override === 'locked' ? [1, 0.75, 1] : 1 }}
          transition={{ duration: 1.8, repeat: Infinity }}
        >
          <span className={`dot ${dotClass}`} />
          {statusText}
        </motion.div>
      </div>
    </motion.header>
  )
}
