import React, { useState } from "react";
import { propertyApi } from "../api/propertyApi";

export function PropertyValuation() {
  const [location, setLocation] = useState("");
  const [size, setSize] = useState("");
  const [bedrooms, setBedrooms] = useState("");
  const [bathrooms, setBathrooms] = useState("");
  const [hasBalcony, setHasBalcony] = useState(false);
  const [valuationResult, setValuationResult] = useState<null | any>(null);
  const [error, setError] = useState("");

  const handleValuation = async () => {
    if (!location || !size || !bedrooms || !bathrooms) {
      setError("Please fill in all required fields");
      return;
    }

    setError("");
    setValuationResult(null);

    const valuationData = {
      location,
      size: parseInt(size),
      bedrooms: parseInt(bedrooms),
      bathrooms: parseInt(bathrooms),
      has_balcony: hasBalcony,
    };

    try {
      const result = await propertyApi.estimateValue(valuationData);
      setValuationResult(result);
    } catch {
      setError("Failed to fetch property valuation");
    }
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <h3 className="text-2xl font-semibold">AI Property Valuation</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <input
          type="text"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          placeholder="Location (city, area)"
          className="px-4 py-2 border rounded-lg w-full"
        />
        <input
          type="number"
          value={size}
          onChange={(e) => setSize(e.target.value)}
          placeholder="Total Area (sq ft)"
          className="px-4 py-2 border rounded-lg w-full"
        />
        <input
          type="number"
          value={bedrooms}
          onChange={(e) => setBedrooms(e.target.value)}
          placeholder="Bedrooms"
          className="px-4 py-2 border rounded-lg w-full"
        />
        <input
          type="number"
          value={bathrooms}
          onChange={(e) => setBathrooms(e.target.value)}
          placeholder="Bathrooms"
          className="px-4 py-2 border rounded-lg w-full"
        />
        <label className="flex items-center space-x-2">
          <input
            type="checkbox"
            checked={hasBalcony}
            onChange={(e) => setHasBalcony(e.target.checked)}
            className="form-checkbox h-5 w-5"
          />
          <span>Has Balcony</span>
        </label>
      </div>
      <button
        onClick={handleValuation}
        className="mt-4 w-full bg-blue-600 text-white px-4 py-3 rounded-lg hover:bg-blue-700"
      >
        Get Valuation
      </button>

      {/* Display Results */}
      {error && <p className="text-red-600 mt-4">{error}</p>}
      {valuationResult && (
        <div className="mt-6 bg-gray-50 p-6 rounded-lg border">
          <h4 className="text-lg font-semibold">Estimated Property Value</h4>
          <p className="text-2xl font-bold text-blue-600">{valuationResult.estimated_value}</p>
          <p>Range: {valuationResult.min_value} - {valuationResult.max_value}</p>
          <p>Confidence: ±{valuationResult.confidence_percentage}%</p>
        </div>
      )}
    </div>
  );
}
