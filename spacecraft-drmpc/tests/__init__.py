# tests/__init__.py
touch tests/__init__.py

# tests/integration/test_complete_system.py
mkdir -p tests/integration
cat > tests/integration/test_complete_system.py << 'EOF'
import pytest
from src.agents.advanced_spacecraft_agent import AdvancedSpacecraftAgent

class TestCompleteSystem:
    @pytest.mark.asyncio
    async def test_full_mission_flow(self):
        agent = AdvancedSpacecraftAgent("test-agent-1")
        result = await agent.execute_mission({
            'phase': 'precision_docking',
            'target': [0, 0, 0],
            'duration': 600
        })
        assert result.success_rate > 0.95
EOF