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
                sh '''
                    # Hentikan dan hapus container lama jika ada
                    if [ $(docker ps -a -q -f name=^/${CONTAINER_NAME}$) ]; then
                        docker stop ${CONTAINER_NAME} || true
                        docker rm ${CONTAINER_NAME} || true
                    fi

                    # Jalankan container baru
                    docker run -d \
                      --name ${CONTAINER_NAME} \
                      --restart always \
                      -p ${HOST_PORT}:${CONTAINER_PORT} \
                      ${IMAGE_NAME}
                '''
            }
        }

        stage('Health Check') {
            steps {
                sleep 5
                sh '''
                    # Jalankan health check dari dalam container
                    docker exec ${CONTAINER_NAME} python -c "import urllib.request; urllib.request.urlopen('http://localhost:${CONTAINER_PORT}/api/health')" || exit 1
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
