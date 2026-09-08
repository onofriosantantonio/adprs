from abc import ABC, abstractmethod
from typing import Any, Dict
from src.utils.logger import get_logger

class BaseAgent(ABC):
    """
    Classe base astratta per tutti gli agenti della pipeline multiagente.
    Ogni agente implementa il metodo `process` che riceve il payload di input
    e arricchisce il dizionario con le proprie feature estratte.
    """

    def __init__(self, agent_id: str, name: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.name = name
        self.config = config
        self.logger = get_logger(f"{self.name} [{self.agent_id}]")

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Elabora i dati e restituisce il dizionario arricchito.
        
        Args:
            data: Contesto della pipeline contenente testi, feature e metadati.
            
        Returns:
            Dict[str, Any] arricchito con le uscite dell'agente.
        """
        pass
