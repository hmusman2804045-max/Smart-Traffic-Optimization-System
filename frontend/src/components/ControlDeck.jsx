import { motion } from 'framer-motion'

const Icon = {
  play: <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>,
  pause: <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z" /></svg>,
  step: <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M5 5v14l8-7zM15 5h3v14h-3z" /></svg>,
  reset: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4"><path d="M4 4v6h6M20 20v-6h-6" /><path d="M20 9A8 8 0 0 0 5.3 6.3L4 10M4 15a8 8 0 0 0 14.7 2.7L20 14" /></svg>,
  green: <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="7" /></svg>,
  bolt: <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2L4 14h6l-1 8 9-12h-6z" /></svg>,
  lock: <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2"><rect x="5" y="11" width="14" height="9" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></svg>,
}

function DeckButton({ color, icon, label, onClick, engaged, disabled }) {
  return (
    <motion.button
      className={`btn ${color}${engaged ? ' engaged' : ''}`}
      onClick={onClick}
      disabled={disabled}
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.96 }}
      style={disabled ? { opacity: 0.4, cursor: 'not-allowed' } : undefined}
    >
      {icon} {label}
    </motion.button>
  )
}

export default function ControlDeck({ running, override, start, pause, stepForward, reset, setOverride }) {
  const locked = override === 'locked'
  return (
    <motion.section
      className="panel"
      initial={{ y: 24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.7, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="panel-head">
        <div className="panel-title"><span className="dot c" /> Control Deck</div>
        <div className="panel-sub">OP-CONSOLE / 01</div>
      </div>
      <div className="panel-body">
        <div className="deck">
          <div className="deck-label">Simulation</div>
          {running ? (
            <DeckButton color="amber" icon={Icon.pause} label="Pause" onClick={pause} />
          ) : (
            <DeckButton color="cyan" icon={Icon.play} label="Start Simulation" onClick={start} disabled={locked} />
          )}
          <DeckButton color="cyan" icon={Icon.step} label="Step Forward" onClick={stepForward} disabled={locked} />
          <DeckButton color="cyan" icon={Icon.reset} label="Reset Data" onClick={reset} />
          <div />

          <div className="deck-label">Manual Override</div>
          <DeckButton
            color="green"
            icon={Icon.green}
            label="Force Green"
            engaged={override === 'green'}
            onClick={() => setOverride('green')}
          />
          <DeckButton
            color="red"
            icon={Icon.bolt}
            label="Emergency Clear"
            engaged={override === 'emergency'}
            onClick={() => setOverride('emergency')}
          />
          <DeckButton
            color="amber"
            icon={Icon.lock}
            label={locked ? 'Unlock Intersection' : 'Lock Intersection'}
            engaged={locked}
            onClick={() => setOverride('locked')}
          />
        </div>
      </div>
    </motion.section>
  )
}
