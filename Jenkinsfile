pipeline {
    agent any

    environment {
        CONTAINER_NAME = 'table-extractor'
        IMAGE_NAME     = 'table-extractor:latest'
        HOST_PORT      = '3022'
        CONTAINER_PORT = '8000'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }

        stage('Deploy Container') {
            steps {
                withCredentials([file(credentialsId: 'table-extractor-env', variable: 'ENV_FILE')]) {
                    sh '''
                        # Hentikan dan hapus container lama jika ada
                        if [ $(docker ps -a -q -f name=^/${CONTAINER_NAME}$) ]; then
                            docker stop ${CONTAINER_NAME} || true
                            docker rm ${CONTAINER_NAME} || true
                        fi

                        # Jalankan container baru dengan file secret .env dari Jenkins
                        docker run -d \
                          --name ${CONTAINER_NAME} \
                          --restart always \
                          --env-file "${ENV_FILE}" \
                          -p ${HOST_PORT}:${CONTAINER_PORT} \
                          ${IMAGE_NAME}
                    '''
                }
            }
        }

        stage('Health Check') {
            steps {
                sleep 5
                sh '''
                    # Jalankan health check dari dalam container dan cetak status
                    docker exec ${CONTAINER_NAME} python -c "
import json, urllib.request
with urllib.request.urlopen('http://localhost:${CONTAINER_PORT}/api/health') as resp:
    data = json.loads(resp.read().decode())
    print('Health Check Response:', data)
    assert data.get('status') == 'ok', 'Healthcheck failed'
" || exit 1
                '''
            }
        }
    }

    post {
        always {
            sh 'docker image prune -f'
        }
    }
}
