import React from "react";
import { Search, Building2, Brain, Star } from "lucide-react";
import { AIFeatures } from "../components/AIFeatures";
import { SearchResults } from "./SearchResults";
import { PropertyValuation } from "../components/PropertyValuation";
import { FeaturedListings } from "../components/FeaturedListings";

export function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* 🔹 Hero Section */}
      <section className="relative h-[600px] flex items-center justify-center">
        <div
          className="absolute inset-0 z-0"
          style={{
            backgroundImage:
              "url(https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&q=80)",
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        >
          <div className="absolute inset-0 bg-black bg-opacity-50"></div>
        </div>

        <div className="relative z-10 text-center px-4">
          <h1 className="text-5xl font-bold text-white mb-6">
            Find Your Dream Home with AI
          </h1>
          <p className="text-xl text-gray-200 mb-8">
            Intelligent real estate recommendations powered by artificial
            intelligence
          </p>

          <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-xl p-2">
            <div className="flex items-center">
              <input
                type="text"
                placeholder="Enter location, property type, or keywords..."
                className="flex-1 px-6 py-3 text-gray-700 focus:outline-none"
              />
              <button className="bg-blue-600 text-white px-8 py-3 rounded-md hover:bg-blue-700 transition duration-200 flex items-center">
                <Search className="h-5 w-5 mr-2" />
                Search
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* 🔹 AI Features Section */}
      <AIFeatures />

      {/* 🔹 Search Results Section */}
      <div className="max-w-7xl mx-auto py-10 px-4">
        <h2 className="text-3xl font-bold text-center mb-6">
          Find & Value Your Dream Home
        </h2>
        <SearchResults />
      </div>

      {/* 🔹 Featured Listings Section */}
      <div className="max-w-7xl mx-auto py-10 px-4">
        <h2 className="text-3xl font-bold text-center mb-6">
          Featured Properties
        </h2>
        <FeaturedListings />
      </div>

      {/* 🔹 Property Valuation Section */}
      <div className="max-w-7xl mx-auto py-10 px-4">
        <PropertyValuation />
      </div>

      {/* 🔹 Features Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            Why Choose RealAI Estate?
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <FeatureCard
              icon={<Brain className="h-8 w-8 text-blue-600" />}
              title="AI-Powered Insights"
              description="Get intelligent property recommendations based on your preferences and market trends."
            />
            <FeatureCard
              icon={<Building2 className="h-8 w-8 text-blue-600" />}
              title="Extensive Listings"
              description="Access thousands of verified properties with detailed information and virtual tours."
            />
            <FeatureCard
              icon={<Star className="h-8 w-8 text-blue-600" />}
              title="Smart Matching"
              description="Our AI matches you with properties that perfectly align with your requirements."
            />
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="bg-gray-50 rounded-xl p-8 text-center transition-transform duration-200 hover:transform hover:scale-105">
      <div className="flex justify-center mb-4">{icon}</div>
      <h3 className="text-xl font-semibold text-gray-900 mb-4">{title}</h3>
      <p className="text-gray-600">{description}</p>
    </div>
  );
}

