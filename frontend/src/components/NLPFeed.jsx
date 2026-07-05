import { motion, AnimatePresence } from 'framer-motion'
import { SENTIMENT } from '../hooks/useSimulation'

export default function NLPFeed({ tweets }) {
  return (
    <motion.section
      className="panel"
      initial={{ y: 24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, delay: 0.25, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="panel-head">
        <div className="panel-title"><span className="dot c" /> Social Intelligence — BERT</div>
        <div className="panel-sub">STREAM / LIVE CLASSIFICATION</div>
      </div>
      <div className="panel-body">
        <div className="feed">
          <AnimatePresence initial={false}>
            {tweets.length === 0 && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-faint)', padding: '18px 6px', textAlign: 'center', letterSpacing: '0.14em' }}
              >
                AWAITING STREAM — START SIMULATION
              </motion.div>
            )}
            {tweets.map((t) => (
              <motion.div
                key={t.id}
                className="tweet"
                layout
                initial={{ opacity: 0, x: -22, filter: 'blur(4px)' }}
                animate={{ opacity: 1, x: 0, filter: 'blur(0px)' }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
              >
                <div style={{ minWidth: 0 }}>
                  <div className="tweet-text">{t.text}</div>
                  <div className="tweet-meta">
                    <span className={`badge b${t.label}`}>{SENTIMENT[t.label]}</span>
                    <span>CONF {(t.confidence * 100).toFixed(1)}%</span>
                    <span>{t.time}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </motion.section>
  )
}
