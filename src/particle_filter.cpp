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

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // 1. Create normal distributions for x, y and theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // 2. Initialize Particles and weights
  for (unsigned int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Create normal distributions for particle x, y and theta
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for (Particle &particle: particles) {
    // Case 1: Yaw == 0. Vehicle driving straight
    if (fabs(yaw_rate) < 0.00001) {
      particle.x += velocity * cos(particle.theta) * delta_t;
      particle.y += velocity * sin(particle.theta) * delta_t;
    }
    // Case 2: Yaw != 0. Vehicle turning
    else {
      particle.x += (velocity / yaw_rate)
                    * (sin(particle.theta + (yaw_rate * delta_t))
                       - sin(particle.theta));
      particle.y += (velocity / yaw_rate)
                    * (cos(particle.theta)
                       - cos(particle.theta + (yaw_rate * delta_t)));
      particle.theta += yaw_rate * delta_t;
    }
    // Add gaussian noise
    particle.x += dist_x(gen);
    particle.y += dist_y(gen);
    particle.theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (LandmarkObs &observation: observations) {
    double min_dist;
    for (unsigned int j = 0; j < predicted.size(); ++j) {
      double distance = dist(observation.x, observation.y, predicted[j].x, predicted[j].y);
      if (j == 0 || (distance < min_dist)) {
        min_dist = distance;
        observation.id = j;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  vector<double> transformed_x;
  vector<double> transformed_y;
  vector<int> transformed_ids;

  for (unsigned int i = 0; i < particles.size(); ++i) {
    std::vector<LandmarkObs> landmarks_predicted;

    // 1. Loop over all landmarks and select the ones in sensor range
    for (const Map::single_landmark_s &landmark: map_landmarks.landmark_list) {
      double distance = dist(particles[i].x,
                             particles[i].y,
                             landmark.x_f,
                             landmark.y_f);
      if (distance < sensor_range) {
        LandmarkObs landmark_predicted;
        landmark_predicted.x = landmark.x_f;
        landmark_predicted.y = landmark.y_f;
        landmark_predicted.id = landmark.id_i;
        landmarks_predicted.push_back(landmark_predicted);
      }
    }

    // 2. Transform the observations to MAP space using Homogenous Transformation
    std::vector<LandmarkObs> transformed_obs;
    for (unsigned int j = 0; j < observations.size(); ++j) {
      LandmarkObs t_obs;

      t_obs.x = particles[i].x
                + cos(particles[i].theta) * observations[j].x
                - sin(particles[i].theta) * observations[j].y;
      t_obs.y = particles[i].y
                + sin(particles[i].theta) * observations[j].x
                + cos(particles[i].theta) * observations[j].y;

      transformed_x.push_back(t_obs.x);
      transformed_y.push_back(t_obs.y);
      transformed_obs.push_back(t_obs);
    }

    // 3. Associate transformed observations with a land mark identifier
    dataAssociation(landmarks_predicted, transformed_obs);

    // 4. Build a vector of landmark ids for the association visualization
    for (unsigned int j = 0; j < transformed_obs.size(); ++j) {
      transformed_ids.push_back(transformed_obs[j].id + 1);
    }

    SetAssociations(particles[i], transformed_ids, transformed_x, transformed_y);

    // 5. Update particle weight
    double weight = 1.0;
    for (LandmarkObs &t_obs: transformed_obs) {
      // calculate normalization term
      double gauss_norm = (1.0 / (2.0 * M_PI * sig_x * sig_y));

      // calculate exponent
      double x_obs = t_obs.x;
      double y_obs = t_obs.y;
      double mu_x = landmarks_predicted[t_obs.id].x;
      double mu_y = landmarks_predicted[t_obs.id].y;

      double exponent = pow(x_obs - mu_x, 2) / (2.0 * pow(sig_x, 2))
                        + pow(y_obs - mu_y, 2) / (2.0 * pow(sig_y, 2));

      // calculate weight using normalization terms and exponent
      weight *= gauss_norm * exp(-exponent);
    }
    particles[i].weight = weight;
    weights[i] = weight;
    //cout << "weight for " << i << "=" << weight << endl;
  }
}

void ParticleFilter::resample() {
  vector<Particle> resampled_particles;
  discrete_distribution<> dist(weights.begin(), weights.end());

  for (unsigned int i = 0; i < particles.size(); ++i) {
    resampled_particles.push_back(particles[dist(gen)]);
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

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
