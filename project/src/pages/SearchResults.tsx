import React, { useState, useEffect } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { Bed, Bath, Square } from "lucide-react";
import { SearchBar } from "../components/SearchBar";
import { propertyApi } from "../api/propertyApi";

type Property = {
  id: string;
  title: string;
  price: number;
  bedrooms: number;
  bathrooms: number;
  square_feet: number;
  images: { url: string }[];
  city: string;
  state: string;
};

export function SearchResults() {
  const [searchParams] = useSearchParams();
  const [properties, setProperties] = useState<Property[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchProperties = async () => {
      setLoading(true);
      const filters = Object.fromEntries(searchParams.entries());
      const results = await propertyApi.search(filters);
      setProperties(results);
      setLoading(false);
    };

    fetchProperties();
  }, [searchParams]);

  return (
    <div className="max-w-7xl mx-auto px-4 py-16">
      <h1 className="text-4xl font-bold text-center mb-10">Find Your Dream Property</h1>

      {/* 🔹 Integrated Search Bar */}
      <SearchBar onSearchResults={setProperties} />

      {/* 🔹 Loading Spinner */}
      {loading && (
        <div className="flex items-center justify-center min-h-screen">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600"></div>
        </div>
      )}

      {/* 🔹 Property Listings */}
      {!loading && properties.length === 0 ? (
        <p className="text-gray-600 text-center mt-6">No properties found.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mt-8">
          {properties.map((property) => (
            <Link key={property.id} to={`/property/${property.id}`} className="bg-white shadow-lg rounded-lg overflow-hidden hover:shadow-xl transition">
              <img
                src={property.images[0]?.url || "/assets/images/default-property.jpg"}
                alt={property.title}
                className="w-full h-56 object-cover"
              />
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
                    <span>{property.square_feet} sqft</span>
                  </div>
                </div>
                <p className="mt-2 text-gray-500">{property.city}, {property.state}</p>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
