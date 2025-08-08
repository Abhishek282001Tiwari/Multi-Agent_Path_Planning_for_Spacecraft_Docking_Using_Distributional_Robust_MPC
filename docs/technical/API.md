# docs/API.md
cat > docs/API.md << 'EOF'
# Spacecraft DR-MPC API Documentation

## Quick Start
```python
from src.agents.advanced_spacecraft_agent import AdvancedSpacecraftAgent

agent = AdvancedSpacecraftAgent("ISS-Dragon-1")
await agent.execute_mission({
    'phase': 'precision_docking',
    'target': [0, 0, 0],
    'features': ['ml', 'formation', 'security']
})
