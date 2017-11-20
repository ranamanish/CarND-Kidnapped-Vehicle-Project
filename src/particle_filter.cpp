/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define EPS 0.001 // Really small number

using namespace std;


// random gaussian
static default_random_engine randomGen;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// number of particles
	num_particles = 500;
	// set the size of particle and weight lists
	particles.resize(num_particles); // Resize the `particles` vector to fit desired number of particles
	weights.resize(num_particles);

	// Create normal distributions for x, y and theta.
	normal_distribution<double> normDist_x(x, std[0]);
	normal_distribution<double> normDist_y(y, std[1]);
	normal_distribution<double> normDist_theta(theta, std[2]);

	// Initialize the particles
	for (int i = 0; i < num_particles; i++){
		particles[i].id = i;
		particles[i].x = normDist_x(randomGen);
		particles[i].y = normDist_y(randomGen);
		particles[i].theta = normDist_theta(randomGen);
		particles[i].weight = 1.0;
	}
	// set initial weights
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	for (int i = 0; i < num_particles; ++i) {
		if (fabs(yaw_rate) < EPS) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		} else {
			const double theta_new = particles[i].theta + yaw_rate * delta_t;

			particles[i].x += velocity / yaw_rate * (sin(theta_new) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate * (-cos(theta_new) + cos(particles[i].theta));
			particles[i].theta = theta_new;
		}

		// Update the particle position with the prediction
		// and add random Gaussian noise
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(randomGen);
		particles[i].y = dist_y(randomGen);
		particles[i].theta = dist_theta(randomGen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	double measDistance;
	double minDistance;

	for (int i = 0; i < observations.size(); ++i) {
		minDistance = 10000000.0 ;
		for (unsigned j = 0; j < predicted.size(); ++j) {
			// Distance between current observed/predicted landmarks
			measDistance = dist(observations[i].x, observations[i].y,
									predicted[j].x, predicted[j].y);

			if (measDistance < minDistance) {
				// update the observation id
				observations[i].id = predicted[j].id;
				minDistance = measDistance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	const double sigmaX = std_landmark[0];
	const double sigmaY = std_landmark[1];
	const double k  = 2 * M_PI * sigmaX * sigmaY;
	double dx = 0.0;
	double dy = 0.0;
	double totalWeight = 0.0;

	for (int i = 0; i < num_particles; i++){
		const double sinTheta = sin(particles[i].theta);
		const double cosTheta = cos(particles[i].theta);
		double measWeight = 0.0;

		// perform the space transformation from vehicle to map
		for (int j = 0; j < observations.size(); j++){
			// Observation measurement transformations
			LandmarkObs observation;
			observation.id = observations[j].id;
			observation.x = particles[i].x + (observations[j].x * cosTheta) - (observations[j].y * sinTheta);
			observation.y = particles[i].y + (observations[j].x * sinTheta) + (observations[j].y * cosTheta);

			bool in_range = false;
			Map::single_landmark_s nearestLandmark;
			double minDistance = 10000000.0;
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {
				Map::single_landmark_s condLandmark = map_landmarks.landmark_list[k];

				double distance = dist(condLandmark.x_f, condLandmark.y_f, observation.x, observation.y);

				if (distance < minDistance) {
					minDistance = distance;
					nearestLandmark = condLandmark;

					if (distance < sensor_range) {
						in_range = true;
					}
				}
			}

			 if (in_range) {
				 dx = observation.x - nearestLandmark.x_f;
				 dy = observation.y - nearestLandmark.y_f;
				 measWeight += dx * dx / sigmaX + dy * dy / sigmaY;
			 }
			 else {
				 measWeight += 100;
			}
		}
		particles[i].weight = exp(-0.5 * measWeight);
		totalWeight += particles[i].weight;
	}

	// Weights normalization to sum(weights)=1
	for (int i = 0; i < num_particles; i++) {
		particles[i].weight /= totalWeight * k;
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    static default_random_engine gen;

    discrete_distribution<> dist_particles(weights.begin(), weights.end());
    vector<Particle> new_particles;
    new_particles.resize(num_particles);
    for (int i = 0; i < num_particles; i++) {
        new_particles[i] = particles[dist_particles(gen)];
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
