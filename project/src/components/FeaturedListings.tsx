import React, { useEffect, useState } from "react";
import { propertyApi } from "../api/propertyApi";
import { Link } from "react-router-dom";
import { Bed, Bath, Square } from "lucide-react";

type Property = {
  id: number;
  title: string;
  location: string;
  price: number;
  total_area: number;
  bedrooms: number;
  bathrooms: number;
  images: { url: string }[];
  city: string;
  state: string;
};

export function FeaturedListings() {
  const [featuredProperties, setFeaturedProperties] = useState<Property[]>([]);

  useEffect(() => {
    async function fetchFeatured() {
      const data = await propertyApi.getFeaturedProperties();
      setFeaturedProperties(data);
    }
    fetchFeatured();
  }, []);

  return (
    <section className="max-w-7xl mx-auto px-6 py-16">
      <h2 className="text-4xl font-bold text-center mb-10">Featured Properties</h2>

      {featuredProperties.length === 0 ? (
        <p className="text-gray-500 text-center">No featured properties available.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {featuredProperties.map((property) => (
            <Link
              key={property.id}
              to={`/property/${property.id}`}
              className="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-xl transition"
            >
              <div className="relative">
                <img
                  src={property.images[0]?.url || "/assets/images/default-property.jpg"}
                  alt={property.title}
                  className="w-full h-56 object-cover"
                />
              </div>

              <div className="p-6">
                <h3 className="text-2xl font-semibold text-gray-900">{property.title}</h3>
                <p className="text-blue-600 text-lg font-bold mt-2">${property.price.toLocaleString()}</p>

                <div className="flex items-center justify-between text-gray-600 mt-3">
                  <div className="flex items-center">
                    <Bed className="h-5 w-5 mr-1" />
                    <span>{property.bedrooms}</span>
                  </div>
                  <div className="flex items-center">
                    <Bath className="h-5 w-5 mr-1" />
                    <span>{property.bathrooms}</span>
                  </div>
                  <div className="flex items-center">
                    <Square className="h-5 w-5 mr-1" />
                    <span>{property.total_area} sqft</span>
                  </div>
                </div>

                <p className="mt-2 text-gray-500">{property.city}, {property.state}</p>
              </div>
            </Link>
          ))}
        </div>
      )}
    </section>
  );
}
