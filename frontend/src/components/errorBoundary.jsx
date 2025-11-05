import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props){ super(props); this.state = { hasError: false, error: null }; }
  static getDerivedStateFromError(error){ return { hasError: true, error }; }
  componentDidCatch(error, info){ console.error("UI crash:", error, info); }
  render(){
    if (this.state.hasError) return (
      <div style={{padding:24}}>
        <h2>Si è verificato un errore nell’interfaccia.</h2>
        <p>Riprova l’azione oppure ricarica la pagina.</p>
      </div>
    );
    return this.props.children;
  }
}
