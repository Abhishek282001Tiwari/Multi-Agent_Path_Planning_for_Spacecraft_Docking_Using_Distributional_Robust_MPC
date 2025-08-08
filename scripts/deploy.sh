# scripts/deploy.sh
cat > scripts/deploy.sh << 'EOF'
#!/bin/bash
echo "🚀 Deploying DR-MPC Spacecraft System..."

# Build and run
docker-compose up --build -d

# Wait for services
echo "⏳ Waiting for services..."
sleep 10

# Test deployment
curl -f http://localhost:8080/health || exit 1

echo "✅ Deployment successful!"
echo "📊 View dashboard: http://localhost:3000"
echo "🔍 API docs: http://localhost:8080/docs"
EOF

chmod +x scripts/deploy.sh